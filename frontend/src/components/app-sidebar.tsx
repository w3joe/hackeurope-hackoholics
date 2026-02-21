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
import { TeamSwitcher } from "./team-switcher";
import {
  AudioWaveform,
  ClipboardList,
  GalleryVerticalEnd,
  LayoutDashboard,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

export function AppSidebar() {
  const pathname = usePathname();

  const teams = [
    {
      name: "dm-Drogerie markt GmbH",
      logo: GalleryVerticalEnd,
      plan: "Germany",
    },
    {
      name: "CVS Inc.",
      logo: AudioWaveform,
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
  ];

  return (
    <Sidebar>
      <SidebarHeader>
        <TeamSwitcher teams={teams} />
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
      <SidebarFooter />
    </Sidebar>
  );
}
