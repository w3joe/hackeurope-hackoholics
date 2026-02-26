"use client";

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import { Activity, ClipboardList, LayoutDashboard } from "lucide-react";
import Image from "next/image";

function CVSLogo({ className }: { className?: string }) {
  return (
    <Image
      src="/CVS_Logo.svg"
      alt="CVS"
      width={32}
      height={32}
      className={className}
    />
  );
}
function DMLogo({ className }: { className?: string }) {
  return (
    <Image
      src="/Dm_Logo.svg"
      alt="dm"
      width={32}
      height={32}
      className={className}
    />
  );
}
import Link from "next/link";
import { usePathname } from "next/navigation";
import { TeamSwitcher } from "./team-switcher";

export function AppSidebar() {
  const pathname = usePathname();

  const teams = [
    {
      name: "dm-Drogerie markt GmbH",
      logo: DMLogo,
      plan: "Germany",
    },
    {
      name: "CVS Inc.",
      logo: CVSLogo,
      plan: "USA",
    },
  ];

  const navigationItems = [
    {
      title: "Dashboard",
      href: "/",
      icon: LayoutDashboard,
      isActive: pathname === "/",
    },
    {
      title: "Orders",
      href: "/orders",
      icon: ClipboardList,
      isActive: pathname.startsWith("/orders"),
    },
    {
      title: "EU Status",
      href: "/status",
      icon: Activity,
      isActive: pathname.startsWith("/status"),
    },
  ];

  return (
    <Sidebar>
      <SidebarHeader>
        <Image
          src={"/VitariskLogo.svg"}
          alt="Vitarisk Logo"
          width={300}
          height={100}
          className="aspect-3/1 max-h-10 w-fit ml-2"
        />
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Navigation</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {navigationItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton
                    render={<Link href={item.href} />}
                    isActive={item.isActive}
                  >
                    <item.icon />
                    <span>{item.title}</span>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
        <SidebarGroup />
      </SidebarContent>
      <SidebarFooter>
        <TeamSwitcher teams={teams} />
      </SidebarFooter>
    </Sidebar>
  );
}
